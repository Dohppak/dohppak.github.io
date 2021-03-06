import React from 'react';
import { StaticQuery, graphql } from 'gatsby';
import PropTypes from 'prop-types';
import Helmet from 'react-helmet';
import { Global } from '@emotion/core';
import { ThemeProvider } from 'emotion-theming';
import 'sanitize.css/sanitize.css';
import 'lato-font/css/lato-font.css';
import Header from './header';
import Footer from './footer';


const minWidthPx = 680;
const maxWidthPx = 960;
const spacingPx = 10;
const centerPadding = `calc((100vw - ${maxWidthPx - (2 * spacingPx)}px) / 2)`;
const smallMedia = `@media(max-width: ${minWidthPx}px)`;
const largeMedia = `@media(min-width: ${maxWidthPx}px)`;
const textColor = '#333';
const accentColor = '#FF9600';

const theme = {
  spacingPx,
  spacing: `${spacingPx}px`,
  headerHeight: '75px',
  textColor,
  accentColor,
  maxWidthPx,
  minWidthPx,
  smallMedia,
  largeMedia,
  centerPadding: {
    padding: `0 ${spacingPx}px`,
    [largeMedia]: {
      paddingLeft: centerPadding,
      paddingRight: centerPadding,
    },
  },
};

const Layout = ({ children }) => (
  <ThemeProvider theme={theme}>
    <Global
      styles={{
        'html, body': {
          width: '100vw',
          height: '100vh',
          margin: 0,
          padding: 0,
          fontFamily: 'Helvetica Neue',
          fontSize: '1em',
          background: '#FFFFFF',
          '*::selection': {
            color: accentColor,
            background: 'none !important',
          },
          '*::-moz-selection': {
            color: accentColor,
            background: 'none !important',
          },
        },
        ul: {
          margin: '0',
        },
        li: {
          marginTop: '0.3em',
        },
        h4: {
          marginBlockEnd: '0em',
        },
        h5: {
          color: 'rgba(0,0,0,0.33)',
        },
        a: {
          textDecoration: 'none',
          fontWeight: 'bold',
          color: textColor,
          transition: 'color 250ms linear',
          ':hover': {
            color: accentColor,
          },
        },
        blockquote: {
          width: '100vw',
          margin: 0,
          background: 'rgba(245,245,245,0.5)',
          ...theme.centerPadding,
          fontFamily: 'Noto Serif KR',
          fontWeight: 'bold',
          fontSize: '1.1em',
          textAlign: 'center',
          padding: `${spacingPx * 2}px`,
        },
      }}
    />
    <StaticQuery
      query={graphql`
        query IndexLayoutQuery {
          site {
            siteMetadata {
              title
              description
            }
          }
        }
      `}
    >
      {({ site: { siteMetadata: site } }) => (
        <main>
          <Helmet>
            <title>
              {site.title}
              {' '}
              &middot;
              {' '}
              {site.description}
            </title>
            <meta charSet="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <meta httpEquiv="X-UA-Compatible" content="IE=edge,chrome=1" />
            <meta name="HandheldFriendly" content="True" />
            <meta name="description" content={site.description} />
          </Helmet>
          <Header />
          {children}
          <Footer />
        </main>
      )}
    </StaticQuery>
  </ThemeProvider>
);

Layout.propTypes = {
  children: PropTypes.element.isRequired,
};

export default Layout;
